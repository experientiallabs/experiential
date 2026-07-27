# `wmo optimize model`: the one-command optimizer (design)

Status: DESIGN, joint-tau master chat, 2026-07-27. Silen ruled the name (`wmo optimize model`)
and the shape (staged, resumable, one upfront forecast, build-grade UX). The existing distill
CLI renames to `wmo optimize distill` (its artifact is an adapter, not the served model; the
rename is recorded in the plan ledger and coordinated with the training chat).

## The promise

After `wmo build`, one command turns the world model into an optimized, servable endpoint:

    wmo optimize model tau-bench

does preflight -> sweep -> fit -> tune -> report, prints one cost forecast up front, asks once,
and ends with the serving snippet plus a three-objective headline (quality / effective cost per
completed task / latency, each vs the fallback anchor). Every stage writes its artifact where
the manual command would have; the orchestrator adds NO new artifact formats, only a manifest.

## Call sites (written first, judged first)

    # day one: routing-only optimization with the registered pool
    wmo optimize model tau-bench

    # with training: distill a student, gate it, add it to the pool, re-sweep it, refit
    wmo optimize model tau-bench --distill distill.toml

    # resume: stages whose inputs are unchanged are skipped with a printed reason
    wmo optimize model tau-bench

    # redo from a stage down (inputs changed out of band, or you want fresh cells)
    wmo optimize model tau-bench --force-from sweep

    # scripted / CI
    wmo optimize model tau-bench --yes --max-usd 25

    # choose the operating point at the end (composes with the D-DIAL contract)
    wmo optimize model tau-bench --cost-quality 0.25

Judgment: obvious, minimal, hard to misuse. No flag is required for the happy path; every flag
names a decision a user actually owns (spend cap, dial point, training config, redo scope).

## Stages

| stage     | wraps                       | artifact (existing formats)                  | cost |
|-----------|-----------------------------|-----------------------------------------------|------|
| preflight | prepare_pool_provider et al | none (report only)                            | free |
| sweep     | `route sweep`               | `<model_dir>/optimize/matrix.json`            | $$   |
| distill?  | `optimize distill run` + `route student` + student-only sweep + matrix merge | run dir + adapter + pool entry + merged matrix | $$$ |
| fit       | `route fit --kind knn`      | `<model_dir>/policy.json` (+ bank sidecar)    | free |
| tune      | `route tune`                | `<model_dir>/policy.json` (+ `policy.base.json`) | free |
| report    | `route report` + scorecard  | `<model_dir>/optimize/report.json`            | ~$0  |

- Artifacts land exactly where `wmo serve` and the manual commands already read/write them; a
  user can drop to any manual command mid-flow and the orchestrator resumes around it.
- CORRECTION (2026-07-27, as built): the tune row above originally named
  `<model_dir>/endpoint.toml`. That file is real (`wmo/serving/endpoint_config.py`,
  `ENDPOINT_CONFIG_FILENAME`; `wmo/serving/server.py:193-210` mounts every policy on the dial its
  `endpoint.toml` asks for), but it is the SERVING-side override, for a policy an operator does
  not want to rewrite: the platform's slider writes it, and `PUT /v1/endpoints/{name}/config`
  persists to it. `wmo optimize route tune` dials the OPTIMIZER's own output instead, rewriting
  `policy.json` in place and preserving the as-fitted bytes in `policy.base.json`. The
  orchestrator's tune stage calls that same function, so the stage single-sources the manual
  command rather than growing a second way to set the same dial.
  The two compose without corrupting each other: `apply_cost_quality` is absolute, not relative
  (it sets all four knobs from the dial alone, recomputes `floor_sim` off the bank, and strips any
  existing dial suffix through `fit_provenance`), and `EndpointRuntime` keeps the on-disk policy
  as `_base_policy`. So an `endpoint.toml` beside a policy this command already dialed lands on
  exactly the artifact it would have produced from the as-fitted one. Migrating the orchestrator's
  dial to write `endpoint.toml` instead is a possible follow-up, not a defect in what shipped.
- A compaction stage slot is RESERVED between sweep and fit (compress-aware sweep + bank refit
  per the representation-consistency seam) and activates when the D-COMPRESS seam (#265) lands;
  designed now so its arrival is additive.
- distill is OPT-IN: it needs TINKER_API_KEY, a config, and real money. Without `--distill` the
  command is routing-only and says so in the plan.

## The plan table (printed before anything spends)

    optimize model: tau-bench                     pool: 5 candidates (.wmo/pool.toml)

      stage      plan                                            est. cost   status
      preflight  resolve 5 backends, check prices                free        ok
      sweep      5 candidates x 20 scenarios x 1 episode         ~$4.20      will run
      fit        knn (guarded, fallback claude-fable-5)          free        will run
      tune       cost_quality 0.25 (balanced)                    free        will run
      report     3-objective headline vs claude-fable-5          ~$0.10      will run

      estimated total ~$4.30 (projection: assumed tokens x real cell counts)
      Proceed? [y/N]

- One confirmation for the whole run (`--yes` skips). Stage-level confirmations inside wrapped
  commands are suppressed by passing their `yes` through; the orchestrator owns consent.
- The estimate line names itself a projection (route sweep discipline, savings.py discipline).
- With `--distill`: the distill stage row shows the DistillConfig's own estimate_run_cost
  projection and the budget cap from `[budget] max_usd`; the total shows both terms.
- `--max-usd` is a global kill: metered spend is checked at stage boundaries and the run stops
  cleanly (resume-able) when the cap would be crossed.

## Resume semantics (the detail that makes it trustworthy)

A manifest `<model_dir>/optimize/optimize-run.json` records, per completed stage: input
fingerprints (pool file hash, scenario-set identity as recorded in the matrix, matrix file hash,
policy file hash, config knobs) and the artifact path + hash it produced. On the next run each
stage is skipped iff its recorded fingerprints match the live inputs, with a printed reason:

      sweep      SKIP (matrix.json is current: same pool, same scenarios, same episodes)

Anything else reruns, and says which input changed. `--force-from <stage>` invalidates that
stage and everything downstream. A stage that fails mid-flight leaves prior artifacts valid;
rerunning resumes at the failed stage. No hidden state: deleting optimize/ resets the manifest
without breaking any manual-command artifact.

## Failure behavior

- Preflight failures are boundary errors before the confirmation, one line per candidate, with
  the fix in the message (route sweep already does this; the orchestrator surfaces it).
- A failed paid stage prints the exact resume command (`wmo optimize model tau-bench`) and, for
  distill, the loop's own `--resume` handoff.
- Uneven sweep coverage keeps route sweep's contract: matrix written, fit withheld, non-zero
  exit, `--allow-uneven-coverage` passthrough.

## Ending (the payoff)

    policy: knn (guarded, fallback claude-fable-5)   dial: 0.25 balanced
    quality  +0.9pt vs fable-5   (WM-simulated, judge rubric-v2/opus-4-8, 20 held-out scenarios)
    cost     -24% effective cost per completed task  (cache-adjusted)
    latency  p50 -0.4s

    serve it:   wmo serve tau-bench
    endpoint:   POST /v1/chat/completions  (model="tau-bench")

Numbers come from the scorecard module (jt/scorecard branch): provenance-labeled, anchor-named,
same-scenario enforced. If the report stage cannot honestly compute a number it prints the
reason, never a blank or a zero.

## Implementation notes

- New `wmo/cli/optimize_model_app.py` command function mounted as `optimize model`; existing
  `model_app` remounts as `optimize distill` (docstrings updated; a one-release alias shim under
  the old name that errors with the new spelling is NOT wanted: the surface is one day old).
- Stage engine is a small typed StagePlan/StageResult pydantic pair + a runner; stages call the
  same library functions the manual commands call (never subprocess re-entry), so consent,
  metering, and artifacts stay single-sourced.
- Matrix merge (for the student-only re-sweep) is a small additive helper on OutcomeMatrix:
  union pools (name-keyed, prices must agree), concat outcomes, reject scenario-id mismatches.
  Named consumer: the distill stage and the ladder benchmark.
- Reuses `wmo/cli/ui.py` rendering; no new colors; tables follow build's style.
