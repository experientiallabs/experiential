# Charter: inverse distillation — train an agent from mined scenarios + the world model

> Proposal for the 48-hour autonomous run. Companion research:
> `../research/scenario-set-construction-sota.md`, `../research/scenario-composition-interpolation-sota.md`,
> and the correction analysis in `../research/scenario-set-e2e-tau-bench.md`.
> Key external reference: World Model Self-Distillation (arXiv 2606.12072) — privileged
> Demonstrator → task-only Executor distillation + RL with verifier feedback.

## Objective

Invert telemetry traces into the three ingredients of training — task distribution (scenario
mining), reward functions (checklists), environment (world model) — and use them to measurably
improve a small student agent, verified against both the mined held-out eval set and the REAL
tau2-bench runner.

## Design (adapted from arXiv 2606.12072's demonstrator/executor split)

1. **Mine** the full tau-bench corpus (~1000 traces) into a verified scenario pool: facets →
   cluster → SemDeDup → synthesize (task + seed state + checklist) → verify (`--drop`).
   Split: train scenarios / held-out eval scenarios, disjoint by source trace (existing splitter).
2. **Privileged teacher collection**: a strong teacher (Gemini via the OpenAI-compatible
   provider endpoint) rolls each train scenario in the world model — *conditioned on the source
   trajectory digest as a hint* (it re-demonstrates rather than solves cold; the paper's
   Demonstrator). k samples per scenario at temperature; keep only checklist-passing trajectories
   (rejection sampling). Balance the kept set by cluster (flattened weights) and force-include
   pinned failure scenarios (recovery demonstrations).
3. **Distill to the executor**: SFT (LoRA) a Qwen student on (task, seed state) → trajectory,
   with NO privileged hint — the paper's Demonstrator→Executor asymmetry. Training on Modal;
   student served back via vLLM behind the OpenAI provider's custom endpoint.
4. **Verify** (the actual deliverable): student-before vs student-after, k=3 passes each, on
   (a) the held-out mined eval scenarios in the world model, and (b) a subset run in the REAL
   tau2 runner (`examples/tau-bench/run_real_scenario.py`) as the sim2real check. Judge =
   family-separated from both the student and the WM (Gemini judges, Nova serves the WM — never
   the same family as what's being graded).
5. **Stretch (only if 1–4 lands early)**: one GRPO round on the student against the WM with
   checklist rewards on ZPD-banded scenarios (pass rate ∈ [0.1, 0.9]); frontier expansion via
   cross-cluster composition, admission-gated by teacher solvability.

## What exists to build on

- `wmh/scenarios/` (this branch): mining, synthesis, verification, `frozen()` eval isolation.
- `feat/rl-arms-sft-ppo-rpp` branch: `examples/tau-bench/rl/export_sft_episodes.py`,
  `merge_adapter_and_splice.py`, `serve_tau_wm.py`, prior SFT-arm results.md — cherry-pick,
  don't rewrite.
- OpenAI provider custom endpoints (PR #67, on main): reaches Gemini AND a Modal vLLM student.
- `RetryingProvider` (main): drop the ad-hoc retry wrapper in `.agents/scripts`.
- Prebuilt tau-bench world model artifact + real tau2 runner for ground truth.

## Success criteria

- Student-after > student-before on held-out mined evals (mean of 3 passes, checklist pass-rate),
  with the delta also directionally confirmed on the real tau2 runner subset.
- Full provenance: every SFT example traces to (scenario_id, source trace_id, teacher rollout).
- Gates clean; PR(s) with honest writeup including negative results if the lift doesn't appear.

## Prior art in-repo: BENCH-B2 (feat/rl-arms-sft-ppo-rpp, results.md)

The direct predecessor already ran: Qwen3.5-9B vs the tau WM — base 55%, trace-SFT (LoRA on 97
raw recorded demonstrations) 60%, PPO/REINFORCE++ flat. Diagnosed SFT failure modes: (1) trained
without think blocks → stopped deliberating; (2) imitates action patterns without checking policy
constraints. **The bar for this run: beat the trace-SFT arm on a comparable held-out protocol,**
attacking exactly those two failure modes — checklists encode the constraints, the privileged
teacher demonstrates *with* reasoning, rejection sampling keeps only constraint-satisfying
trajectories, and cluster balancing fixes the demonstration-coverage skew (their SFT lift
concentrated in airline, where its demos lived). Mind the D32/D35 telecom-leakage caveat: report
per-domain, read non-telecom columns for signal.

## Execution notes (2026-07-03, autonomous run)

- Box: azureuser@4.154.170.26 (h100-dev-box, 2×H100). GPU 0 runs the user's `qwen35pilot` vLLM
  (Qwen3.5-9B, port 8000) — read-only for me (student-before baseline endpoint). GPU 1 free for
  training. Disk tight (~5GB free): LoRA-only, no full checkpoints.
- Stack verified: Nova Lite WM (Bedrock), Gemini 2.5 Pro teacher + Flash judge via the OpenAI
  provider custom endpoint (`WMH_ENDPOINT_API_KEY`), claas-verl tau scaffold pattern for the
  student loop.
- Mining (stage 1) launched over all 1033 traces: 822 train / 211 eval by stable hash; facet
  outcomes {success 807, failure 63, unknown 163}.

## Known risks

- WM fidelity bounds everything: run open-loop fidelity on the on-policy distribution before/after;
  real-runner check is the backstop.
- Reward hacking / judge leniency: family separation + checklist decomposition + spot-read
  transcripts (AGENTS rule 12).
- Verdict variance: k=3 everywhere, report std.
- Modal cold starts / GPU quota: budget wall-clock for training ≤ 6h; fall back to a smaller
  student (Qwen 4B-class) if 9B training is too slow.
