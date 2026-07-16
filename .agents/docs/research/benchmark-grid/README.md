# benchmark-grid — reference run inputs & artifacts

The inputs and saved results of the 4-benchmark × 5-model `wmh eval grid` reference run
(terminal-tasks, tau-bench, kimi-gui-control, swe-bench). The user-facing feature doc is
`docs/reference/eval_grid.md`; this directory is research scratch (raw JSONs + evolved prompts),
not a product artifact.

## ⚠️ Read before citing any number here

**1. The archived JSONs are `rubric-v1`, reference-only.** Every `grid_*_{api,qwen}.json` was
scored by the pre-overhaul judge (before PR #83). They are stamped `"judge_version": "rubric-v1"`.
rubric-v1 scores run roughly +0.12 higher than the current `rubric-v2` judge on the same
predictions, so **these numbers are NOT comparable to any fresh run** (`wmh eval grid` now stamps
`rubric-v2`). Re-run under `rubric-v2` before comparing.

**2. Eight of the twenty GEPA prompts are byte-identical to `BASE_ENV_PROMPT` (no-op GEPA).**
Those GEPA runs found no improvement and returned the base prompt verbatim. In the archived JSONs
their `gepa` / `gepa_rag` cells are therefore just re-runs of `base` / `base_rag`, and any
difference between the two is **judge/sampling noise, not GEPA lift**. The eight no-op cells:

| benchmark | model |
|---|---|
| kimi-gui-control | Qwen-AgentWorld |
| swe-bench | GPT-5.4 Mini |
| swe-bench | Qwen-AgentWorld |
| tau-bench | Opus 4.8 |
| tau-bench | Qwen-AgentWorld |
| terminal-tasks | GPT-5.5 |
| terminal-tasks | Haiku 4.5 |
| terminal-tasks | Qwen-AgentWorld |

The other twelve prompts are genuinely evolved. Do not read a `+GEPA` delta for any row above as a
result. `run_grid` now **skips** the GEPA cells of a base-identical prompt (treats it as "no evolved
prompt"), so fresh runs cannot reintroduce these no-op cells — the disclosure here is only about the
`rubric-v1` snapshot preserved in this directory.

## Contents

- `gepa-prompts/<suite>/<Label>.txt` — the 20 evolved prompts fed to the reference run (8 no-op,
  12 evolved, per above).
- `grid_<suite>_{api,qwen}.json` — the saved `rubric-v1` `GridResult`s (API-model grid + the
  self-hosted Qwen grid, which runs in its own process). Re-render figures from these with
  `wmh eval grid-plot` / `grid-heatmap` (charts are regenerable, so no PNGs are committed).

One more reference-run caveat: for `swe-bench` the api and qwen JSONs were scored on slightly
different held-out sets (api = 8 traces / 12 steps, qwen = 5 traces / 17 steps) despite the same
split config, because they were captured in separate passes. `grid-plot` merges them with a `max`
over the counts, so a combined swe-bench chart's "held-out traces / judged steps" subtitle is an
upper bound, not a per-bar count. This is another reason these archives are reference-only.

The `kimi-gui-control` trace corpus that this run scored now lives at
`packages/environment-capture/kimi-gui-control/` (Hub-hosted, not committed) — see its README.
