# tau-bench real defaults: routing measured on Sierra's actual benchmark

## Why real-only

Every prior tau conclusion here was measured inside a world model and scored by an LLM judge. A WM-vs-real probe then inverted the simulated headline: the world-model judge underscored fable-5's correct INACTION by roughly 33 points, and in reality fable-5 was winning. So every wm_simulated tau finding, opus-5's apparent dominance included, was a hypothesis this matrix re-tested. It does not replicate: on real episodes fable-5 leads (0.757 unpaired mean), glm-5.2 is second (0.737), opus-5 is THIRD (0.730). The served model dir .wmo/models/tau-bench has default_model and guard_model = opus-5, fitted on the world-model matrix; that policy's central choice is not supported by the real benchmark.

## Protocol

Pinned eval split loaded from rl/scenarios_eval.jsonl (seed 4405, the whole test split), never re-derived: 20 scenarios over airline 7, retail 8, telecom 5. Scenario ids are `<domain>:<task_id>` because airline and retail both number from 0 and all 50 airline ids collide with retail ids. Cohort `turns100-t1800-tok8192-r0-sim-gpt-5.4-mini`: max_turns 100, per-episode timeout 1800s, max_tokens 8192, user simulator azure/gpt-5.4-mini, tau2's own retries 0 with retry owned by the runner. All five forwarded explicitly because tau2's defaults differ; rows record the label and consumers refuse to pair across labels. The user simulator is ENVIRONMENT, one cheap model identical for every run, because letting it vary per candidate would change the environment per candidate. Judge: tau2's own reward, read from its results.json; this leg never runs a wmo judge.

FIFTEEN candidates, not sixteen: the pinned user simulator is excluded from the candidate list by design, and the canonical user sim IS gpt-5.4-mini. Right methodologically, and it costs tau its cheapest frontier arm, so tau's arm count is not comparable to the sibling legs' unless their harnesses pin a model out of the pool the same way.

## Results

Paired per scenario against anchor fable-5 ($1.3494 per completed task), cache-adjusted effective cost per COMPLETED task from wmo.optimize.scorecard, cluster-bootstrap CI95, exact sign test:

    candidate         cost %        cost CI95     quality pt     quality CI95      p     n
    glm-5.2           -88.1   [-90.9, -84.4]          -2.63   [-23.7, +18.4]  1.000    19
    sonnet-5          -62.8   [-72.3, -48.9]          -2.63   [-23.7, +21.1]  1.000    19
    opus-5            -49.3   [-63.4, -31.4]          -2.78   [-27.8, +22.2]  1.000    18
    gpt-5.6-sol       -68.3   [-76.6, -56.0]          -5.26   [-28.9, +18.4]  0.754    19
    gpt-5.6-terra     -85.5   [-88.9, -81.5]          -7.89   [-28.9, +13.2]  0.727    19
    qwen3.6-27b       -96.3   [-97.3, -94.7]          -7.89   [-28.9, +13.2]  0.727    19
    kimi-k3           -73.8   [-80.9, -63.4]         -10.53   [-34.2, +15.8]  0.388    19
    opus-4-8          -42.5   [-59.9, -15.5]         -10.53   [-34.2, +13.2]  0.754    19
    kimi-k2.6         -91.1   [-93.2, -87.5]         -13.16   [-31.6,  +5.3]  0.219    19
    deepseek-v4-pro   -82.2   [-87.1, -75.4]         -15.79   [-39.5,  +7.9]  0.388    19
    gpt-5.5           -61.5   [-72.3, -41.0]         -21.05   [-44.7,  +2.6]  0.227    19
    qwen3.5-9b        -99.0   [-99.3, -98.6]         -29.17    [-54.2, -4.2]  0.125    12
    gpt-5.6-luna      -91.2   [-94.1, -84.3]         -36.84   [-57.9, -15.8]  0.012    19
    haiku-4-5         -84.9   [-89.4, -74.5]         -39.47   [-63.2, -15.8]  0.006    19

Routed rungs, LOO-CV over all 20 scenarios (refit per fold on the other 19, route only the held-out one), which is the properly powered reading:

    dial 0.00 (quality-max)   cost +20.3%   quality -28.95 pt   p=0.016
    dial 0.25 (balanced)      cost +16.6%   quality -28.95 pt   p=0.016
    dial 0.50                 cost -41.8%   quality -26.32 pt   p=0.039
    dial 0.75                 cost -49.0%   quality -23.68 pt   p=0.070
    dial 1.00 (max-savings)   cost -46.9%   quality -21.05 pt   p=0.125

Balanced and quality-max are NOT SHIPPABLE here: worse on both axes. Max-savings is a real tradeoff but strictly worse than serving glm-5.2. The single split (n=6) said -83% at -25 pt with p=0.500, which is what an underpowered held-out band looks like and why the LOO leg was required. MECHANISM: fit instability. fable-5 and glm-5.2 differ by 2.63 points, far below noise, so which is "best single" flips between folds, and fable-5-heavy fold policies route away from fable-5 exactly where it was right.

## Caveats that travel with every number

1. POWER IS ASYMMETRIC. Cost CIs are about +-3%; quality CIs about +-20 POINTS at 19-20 paired scenarios. "-2.63 pt, p=1.000" does NOT establish parity, it establishes that 20 points would be undetectable. The claim is large, well-resolved savings with quality UNRESOLVED. The corpus does discriminate large gaps (haiku-4-5, gpt-5.6-luna above).
2. 7 of 20 tasks are NOT deterministic: they carry NL_ASSERTION in their reward_basis, scored by tau2's own NL-assertion judge. Rows record which.
3. THE gpt-5.6 FAMILY RAN WITH REASONING OFF. All three refuse function tools any other way ("Function tools with reasoning_effort are not supported ... use /v1/responses or set reasoning_effort to 'none'"). Reasoning off is the only configuration in which they complete an episode AND the only one our serving path can reach, so these rows measure what a customer could route to. They are not comparable to a reasoning-on number.
4. A REWARD OF 1.0 DOES NOT ALWAYS MEAN THE SAME THING. Several pinned airline tasks are correct-INACTION tasks: policy forbids the refund, the right behavior is to leave the database alone, and the DB check passes for any agent that does nothing. tau2 additionally reports COMMUNICATE with "No communicate_info to evaluate" while still contributing full credit. Rows carry reward_basis, reward_breakdown and the vacuous components so earned credit can be separated from credit the task handed out. This is the WM judge's blind spot appearing on the real benchmark from the other side.
5. UNSCORED AND MISSING, named not hidden. 6 of 600 cells never produced a row: the two hardest telecom [mms_issue] scenarios on fable-5 (x3), opus-5 (x2) and gpt-5.5, whose episodes hit the batch kill deadline twice. 20 more unscored: 25 total empty-assistant-message failures (overwhelmingly qwen3.5-9b, which lost 21 of 40 and reproduced identically on retry) and 3 provider ContentPolicyViolationError. qwen3.5-9b reads 0.632 over survivors and 0.300 counting its own failures; the second is fair. Nothing was imputed into stored rows.

## Statistics

Paired per scenario over the intersection scored on both sides, never unpaired means. Cluster-bootstrap CIs on both axes. Noise floor +-0.015 to 0.02 mean reward, drawn on every delta chart; no finding headlines a delta inside it. Every series names its judge and provenance; no delta crosses provenance. Dial meanings from wmo.optimize.knn.apply_cost_quality: 0.0 quality-max, 0.25 balanced (shipped default), 1.0 max-savings, and past 0.25 the dial is a decision to spend less rather than a free lunch.

## Reproduction

    .agents/scripts/run_tau_bench_defaults_grid.sh          # the grid, exactly as run
    .agents/scripts/run_tau_bench_defaults_smoke.sh         # harness validation, all 5 provider families
    wmo optimize route fit <matrix> --kind knn --embedder auto --rag-num 7 --min-pairs 2 --z 0.5 --floor-q 0.05
    wmo optimize route report <matrix> <policy> --baseline fable-5 --endpoint tau-bench \
      --provenance real_episode --judge "tau2 reward (7/20 pinned tasks include tau2's NL-assertion judge)" \
      --scenario-label "on the pinned tau2-bench eval split (20 scenarios, real benchmark episodes)"
    uv run --extra viz python .agents/docs/research/corners/common/build_corners.py \
      --lens bench-defaults-tau --anchor fable-5 --loo --out-dir .agents/docs/research/bench-defaults/tau/figures
    .agents/scripts/tau_real_run_artifacts.py then wmo runs backfill   # run history + costs
