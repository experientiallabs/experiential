"""Emit gen/report.md, the Bench-GEN standing report artifact (design doc: reports ship per step).

Narrative + numbers are hand-authored from this run's metrics.json / verification_report.json /
tau_scenarios.json; re-run after a fresh build to refresh the numbers, then update the prose.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(".agents/docs/research/gev_bench_results/gen/report.md")

REPORT = """# Bench-GEN: scenario generation, first empirical run

Corpus: tau-bench, first 100 traces (deterministic cap), budget 15. Run date 2026-07-23.
All models on Bedrock us-east-1. Facets, cluster naming, synthesis, and the back-agreement gate
ran on Opus 4.8 (pinned via `--provider bedrock --model claude-opus-4-8`). Solvability rollouts ran
the agent and the world-model simulator on Haiku 4.5 (the world model's serve provider); the
checklist grader ran on Opus 4.8 (the `judge` role in `.wmh/settings.toml`). Facet clustering used
the offline HashingEmbedder (repo default, lexical, no cost).

## Headline numbers

- Scenarios produced: 9 (budget 15).
- Corpus coverage: 39 percent of the 100-trace facet set within cosine tau=0.7 of a selection.
- Clusters: 10 mined; scenarios landed in only 5.
- Back-agreement: 100 percent (9/9).
- Solvable: 44 percent (4/9).
- Seed-state health: 9/9 scenarios have an EMPTY structured seed state; 9/9 have a populated
  scratchpad.
- All 9 survivors are pinned-failure scenarios (source outcome = failure). This is the central
  finding; see below.

## Why only 9 of budget 15 survived, and why two mid-size clusters got zero

The 6 missing scenarios are NOT dedup and NOT allocation rounding. Two mechanisms in
`wmh/scenarios/mining/selection.py` and `wmh/scenarios/builder.py` combine:

1. **Failure pinning consumed the budget.** `_pin_failures` (selection.py) guarantees every failure
   category present in the corpus keeps at least one exemplar, and it does so by EVICTING the
   lowest-weight unpinned selection for each uncovered category. The first-100 corpus has 14 failure
   traces, each a distinct failure category (wrong_action_upgrade_not_cancel, wrong_item_returned,
   policy_violation_refused, refund_to_wrong_payment_method, unresolved_transfer,
   unsupported_request, policy_not_eligible, wrong_action_cancel_vs_return,
   duplicate_bookings_instead_of_modify, and more). The proportional and coverage passes first pick
   cluster medoids, which are almost all success traces (86 of 100 traces succeed). Failure pinning
   then evicts up to 14 of those success picks to seat one exemplar per failure category. With 14
   categories competing for 15 slots, the failure pins crowd out nearly every success pick. That is
   why the large, success-heavy clusters "Product exchange requests" (15 percent of the corpus) and
   "Pending order modifications" (14 percent) received zero scenarios: their medoids are success
   traces, they were selected, and then evicted by failure pinning.

2. **The back-agreement gate then dropped 6.** In `builder.py`, `validate_checklists=True` grades
   each synthesized checklist against its own source trajectory; a checklist that misgrades its
   source is regenerated once, then dropped. Six of the fifteen selections failed this gate and were
   dropped, leaving 9. Every survivor is a pinned failure, so the near-total failure skew from
   step 1 carried all the way through.

Net effect: the eval set is 9 failure-derived scenarios, and weight renormalization then hands them
the full traffic mass. That INVERTS the "eval mirrors traffic" intent (design doc / selection.py
docstring): an 86-percent-success corpus produced a 100-percent-failure eval. This is the same
value-function concern as design principle 1, but flipped: the rare tail (failures) is not just
preserved, it has crowded out the head. On a corpus with many distinct failure categories relative
to the budget, failure pinning dominates. Suggested follow-ups: cap the fraction of the budget
pinning may claim, or raise the budget well above the failure-category count, and surface the
selected-vs-dropped counts (today the CLI prints only the final total, so a 40 percent build-time
drop and a 100-percent-failure skew are both invisible).

## Back-agreement 9/9 vs solvable 4/9: reading the failing rollouts

Back-agreement is 100 percent by construction, not as independent evidence: the build already drops
any scenario whose checklist misgrades its own source, so the survivors pass back-agreement
tautologically. The verify-time back-agreement re-confirms it. The real back-agreement signal was
the 6 build-time drops.

The 5 unsolvable rollouts fall into three causes, and only one is a scenario defect:

- **Missing simulated user (harness limitation, dominant).** tau-bench tasks are multi-turn
  user-agent DIALOGUES ("Ask the agent to tell you each flight's duration so you can decide";
  "communicate this to Emma"). The solvability leg rolls a single autonomous agent against the world
  model with NO user simulator, so any criterion requiring "confirm with the user", "communicate to
  the user", or "ask the user" fails structurally. `wrong_action_upgrade_not_cancel` (pass_rate 0.0,
  "cancelled ... without any user interaction, confirmation, or communication") and
  `policy_not_eligible` (pass_rate 0.75, "correctly determined ineligibility but ... no message
  clearly communicating this to Emma") are both this. These are not scenario defects and not
  simulator errors; they are a mismatch between a conversational benchmark and a single-agent
  rollout.
- **World-model (simulator) confusion.** `wrong_item_returned` (pass_rate 0.4): the agent issued the
  keyboard-return call but "the observation shows a Water Bottle and doesn't confirm the keyboard
  item." The Haiku-served world model returned the wrong item state. This is the simulator fumbling,
  as expected when a cheap model reconstructs a stateful catalog from prose. A real WM-fidelity
  limit, not a task defect.
- **Agent capability / checklist strictness.** `unresolved_transfer` (pass_rate 0.75): the outcome
  was correct (Silver tier, 2 bags) but a process criterion ("confirmed retrieval identifying
  economy") was not demonstrated because the agent took a shortcut; the judge's own instruction is
  "judge outcomes, not mechanics", so this is a mild checklist over-specification.
  `duplicate_bookings_instead_of_modify` (pass_rate 0.0): the agent gathered data but never acted,
  a Haiku-agent capability miss compounded by the missing user loop.

Conclusion: the 44 percent solvable rate mostly reflects the rollout harness (no user simulator,
Haiku agent and Haiku simulator) rather than scenario quality. Do NOT prune scenarios on this run's
solvability.

## Solvable semantics (do not misread the column)

From `wmh/scenarios/verification/verify.py` and `verification/judge.py`:
`solvable = rollout.success`, where `success` is the judge's HOLISTIC boolean ("true if the episode
as a whole achieved the task"), returned by the LLM independently of the per-item `passed` list.
`rollout_pass_rate = sum(passed) / len(passed)` is the fraction of checklist items marked passed.
They are different axes: a rollout can pass 3 of 4 checklist items (pass_rate 0.75) while the judge
still sets `success=false` because the task as a whole was not achieved (e.g. the user was never
informed, or no action was taken). So `rollout_pass_rate 0.75` with `solvable false` is expected and
correct, not a bug. `ScenarioVerdict.ok` = `solvable and back_agreement is not False`.

## seed_state_health: 9/9 empty structured state

By-design for this corpus, not a synthesis defect. The synthesizer seeds `seed_state.structured`
from the source trace's `wmh.state.structured` span attribute; the tau-bench OTel capture never
recorded machine-readable env state on its action spans (only the free-text scratchpad is derived),
so there is nothing structured to seed from. The scratchpad is populated on all 9.

Consequence for the execute step: solvability rollouts (and any future Bench-EXEC run on this
corpus) start from a PROSE-only description of the world. The world model must reconstruct account
balances, reservation records, and order catalogs from the scratchpad narrative, which is exactly
where the `wrong_item_returned` simulator confusion came from. For execute-step realism this corpus
is limited; a capture that records `wmh.state.structured` (or a corpus like bird-sql with a real
backing store) is needed to test structured-state grounding. This is a corpus-capture gap to log,
not a bug in synthesis or verification.

## Other automated metrics

- Cluster naming is degenerate under the lexical embedder: near-duplicate cluster names ("Order
  modification and returns" vs "Order Returns and Cancellations" vs "Order tracking and returns";
  "Flight reservation management" vs "Airline reservation management" vs "Flight Reservation
  Modifications"). HashingEmbedder clusters by wording, so paraphrases of one intent fragment across
  clusters, inflating the cluster count and fragmenting allocation. A semantic embedder should be
  A/B'd before the coverage number is trusted.
- Weight distribution over the 9 survivors: min 0.036, max 0.132, mean 0.111, sum 1.0.
- Checklists are 4 to 5 items each, all observable post-conditions; no empty checklists.

## Cost and wall time

- Smoke build (5 traces, budget 3): 33 s, 2 scenarios (1 dropped).
- Full build (100 traces, budget 15): 100 s.
- World model `gev-tau` build (fidelity low, Haiku serve, all 1033 corpus traces): 5.5 s, 0 LLM
  calls (low fidelity skips GEPA; pure-RAG simulator).
- Verification (9 scenarios, max_steps 10): 209 s.
- Total wall: about 6 minutes.
- Dollar cost: NOT surfaced. `wmh scenarios build` / `verify` do not route through the metered
  provider, so no `.wmh/runs` entry is written and no cost line prints. Adding cost accounting to
  these CLIs is a suggested follow-up.

## Files

- `smoke_scenarios.json` : 2-scenario smoke build.
- `tau_scenarios.json` : the 9-scenario built ScenarioSet.
- `verification_report.json` : per-scenario back-agreement + solvability verdicts and critiques.
- `metrics.json` : all automated metrics (machine-readable).
- `labeling_sheet.md`, `labels_template.jsonl` : the blind manual leg.

## Reproduce

```bash
cd ~/Desktop/Projects/wmh-gev-bench   # worktree on branch research/gev-benchmarks

# Corpus: packages/environment-capture/tau-bench/traces.otel.jsonl is an untracked local capture
# (10578 span-lines -> 1033 traces); copied from the main checkout at the same path.

# 1. Build the scenario set (100 traces, budget 15).
uv run wmh scenarios build \
  --file packages/environment-capture/tau-bench/traces.otel.jsonl \
  --limit 100 --budget 15 \
  --provider bedrock --model claude-opus-4-8 --region us-east-1 \
  --out .agents/docs/research/gev_bench_results/gen/tau_scenarios.json

# 2. Build a low-fidelity world model for solvability rollouts (Haiku serve).
uv run wmh build --name gev-tau --no-interactive \
  --file packages/environment-capture/tau-bench/traces.otel.jsonl \
  --provider bedrock --model claude-haiku-4-5 --judge-model claude-haiku-4-5 \
  --region us-east-1 --fidelity low --root .wmh </dev/null

# 3. Verify (back-agreement + solvability), persisting the report JSON. Judge role (Opus 4.8)
#    is set in .wmh/settings.toml [models.judge]; agent + simulator use the WM's Haiku serve.
uv run python .agents/scripts/gev_bench/run_verify.py \
  --scenarios .agents/docs/research/gev_bench_results/gen/tau_scenarios.json \
  --file packages/environment-capture/tau-bench/traces.otel.jsonl \
  --name gev-tau --max-steps 10 \
  --out .agents/docs/research/gev_bench_results/gen/verification_report.json

# 4. Automated metrics + manual-leg artifacts.
uv run python .agents/scripts/gev_bench/build_report.py \
  --scenarios .agents/docs/research/gev_bench_results/gen/tau_scenarios.json \
  --file packages/environment-capture/tau-bench/traces.otel.jsonl \
  --report .agents/docs/research/gev_bench_results/gen/verification_report.json \
  --outdir .agents/docs/research/gev_bench_results/gen

# 5. This report.
uv run python .agents/scripts/gev_bench/write_report.py
```
"""


def main() -> None:
    assert "—" not in REPORT, "no em dashes allowed"
    OUT.write_text(REPORT, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
